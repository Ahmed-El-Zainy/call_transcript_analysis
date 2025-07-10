import logging
import os
import yaml
import json
from datetime import datetime
import sys
from typing import Dict, Any, Optional, List
from pathlib import Path
import threading
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import gzip
import shutil
from enum import Enum

class LogLevel(Enum):
    """Enum for log levels to provide better type safety."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

class LogFormat(Enum):
    """Predefined log formats."""
    SIMPLE = "%(levelname)s - %(message)s"
    STANDARD = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DETAILED = "%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(funcName)s - %(message)s"
    JSON = "json"  # Special case for JSON formatting

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'filename': record.filename,
            'line_number': record.lineno,
            'function': record.funcName
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        # Add extra fields if present
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'stack_info', 'exc_info', 'exc_text']:
                log_entry[key] = value
                
        return json.dumps(log_entry)

class UniversalLogger:
    """Enhanced universal logger that can be used across any project."""
    
    def __init__(self, config_path: Optional[str] = None, project_name: Optional[str] = None):
        """Initialize the universal logger with configuration."""
        self.project_name = project_name or "default_project"
        self.config = self._load_config(config_path)
        self.loggers = {}
        self.base_log_dir = self.config.get('base_log_dir', 'logs')
        self._setup_base_directory()
        self._lock = threading.Lock()
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML or JSON file."""
        default_config = {
            'base_log_dir': 'logs',
            'default_level': 'INFO',
            'format': 'STANDARD',
            'console_output': True,
            'file_output': True,
            'rotation': {
                'enabled': True,
                'type': 'size',  # 'size' or 'time'
                'max_bytes': 10485760,  # 10MB
                'backup_count': 5,
                'when': 'midnight',  # for time-based rotation
                'interval': 1,
                'compress': True
            },
            'filters': {
                'exclude_patterns': [],
                'include_patterns': [],
                'min_level_override': {}
            },
            'context': {
                'add_hostname': True,
                'add_process_info': True,
                'add_thread_info': True,
                'custom_fields': {}
            },
            'modules': {}
        }
        
        if not config_path:
            return default_config
            
        config_file = Path(config_path)
        if not config_file.exists():
            return default_config
            
        try:
            with open(config_file, 'r') as file:
                if config_file.suffix.lower() == '.json':
                    loaded_config = json.load(file)
                else:
                    loaded_config = yaml.safe_load(file)
                    
            # Merge with default config
            default_config.update(loaded_config)
            return default_config
            
        except Exception as e:
            print(f"Error loading config file: {e}. Using default configuration.")
            return default_config

    def _setup_base_directory(self):
        """Setup the base directory structure for logs."""
        log_dir = Path(self.base_log_dir) / self.project_name
        log_dir.mkdir(parents=True, exist_ok=True)

    def _get_log_path(self, module_name: str, create_hierarchy: bool = True) -> Path:
        """Generate the hierarchical path for log files."""
        base_path = Path(self.base_log_dir) / self.project_name
        
        if create_hierarchy and self.config.get('hierarchical_structure', True):
            now = datetime.now()
            log_path = base_path / str(now.year) / f"{now.month:02d}" / f"{now.day:02d}"
            log_path.mkdir(parents=True, exist_ok=True)
            return log_path / f"{module_name}.log"
        else:
            return base_path / f"{module_name}.log"

    def _create_formatter(self, format_type: str) -> logging.Formatter:
        """Create formatter based on format type."""
        if format_type == "JSON":
            return JSONFormatter()
        elif hasattr(LogFormat, format_type):
            return logging.Formatter(LogFormat[format_type].value)
        else:
            # Custom format string
            return logging.Formatter(format_type)

    def _add_context_filter(self, logger: logging.Logger):
        """Add context information to log records."""
        class ContextFilter(logging.Filter):
            def __init__(self, config):
                super().__init__()
                self.config = config
                
            def filter(self, record):
                if self.config.get('add_hostname', True):
                    import socket
                    record.hostname = socket.gethostname()
                    
                if self.config.get('add_process_info', True):
                    record.process_name = os.path.basename(sys.argv[0])
                    
                if self.config.get('add_thread_info', True):
                    record.thread_name = threading.current_thread().name
                    
                # Add custom fields
                for key, value in self.config.get('custom_fields', {}).items():
                    setattr(record, key, value)
                    
                return True
                
        logger.addFilter(ContextFilter(self.config.get('context', {})))

    def _create_file_handler(self, log_path: Path, module_config: Dict[str, Any]) -> logging.Handler:
        """Create file handler with rotation if enabled."""
        rotation_config = self.config.get('rotation', {})
        
        if not rotation_config.get('enabled', True):
            return logging.FileHandler(log_path)
            
        if rotation_config.get('type') == 'time':
            handler = TimedRotatingFileHandler(
                log_path,
                when=rotation_config.get('when', 'midnight'),
                interval=rotation_config.get('interval', 1),
                backupCount=rotation_config.get('backup_count', 5)
            )
        else:
            handler = RotatingFileHandler(
                log_path,
                maxBytes=rotation_config.get('max_bytes', 10485760),
                backupCount=rotation_config.get('backup_count', 5)
            )
            
        # Add compression if enabled
        if rotation_config.get('compress', True):
            def compress_rotated_file(source, dest):
                with open(source, 'rb') as f_in:
                    with gzip.open(dest + '.gz', 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(source)
                
            handler.rotator = compress_rotated_file
            
        return handler

    def get_logger(self, module_name: str, **kwargs) -> logging.Logger:
        """Get or create a logger for a specific module."""
        with self._lock:
            if module_name in self.loggers:
                return self.loggers[module_name]

            # Create new logger
            logger = logging.getLogger(f"{self.project_name}.{module_name}")
            
            # Prevent duplicate handlers
            if logger.handlers:
                logger.handlers.clear()
                
            # Get module-specific configuration
            module_config = self.config['modules'].get(module_name, {})
            
            # Set log level
            level_name = module_config.get('level', self.config['default_level'])
            if isinstance(level_name, str):
                level = getattr(logging, level_name.upper())
            else:
                level = level_name
            logger.setLevel(level)

            # Create formatter
            format_type = module_config.get('format', self.config.get('format', 'STANDARD'))
            formatter = self._create_formatter(format_type)

            # Add file handler if enabled
            if self.config.get('file_output', True):
                log_path = self._get_log_path(module_name)
                file_handler = self._create_file_handler(log_path, module_config)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)

            # Add console handler if enabled
            if self.config.get('console_output', True):
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)

            # Add context filter
            self._add_context_filter(logger)

            self.loggers[module_name] = logger
            return logger

    def create_child_logger(self, parent_module: str, child_name: str) -> logging.Logger:
        """Create a child logger that inherits from parent."""
        return self.get_logger(f"{parent_module}.{child_name}")

    def set_level_for_all(self, level: str):
        """Set log level for all existing loggers."""
        log_level = getattr(logging, level.upper())
        for logger in self.loggers.values():
            logger.setLevel(log_level)

    def add_custom_handler(self, module_name: str, handler: logging.Handler):
        """Add a custom handler to a specific module logger."""
        logger = self.get_logger(module_name)
        logger.addHandler(handler)

    def log_performance(self, module_name: str, operation: str, duration: float, **kwargs):
        """Log performance metrics in a structured way."""
        logger = self.get_logger(module_name)
        perf_data = {
            'operation': operation,
            'duration_ms': duration * 1000,
            'performance_metric': True,
            **kwargs
        }
        
        # Add performance data to log record
        logger.info(f"Performance: {operation} took {duration:.4f}s", extra=perf_data)

    def log_error_with_context(self, module_name: str, error: Exception, context: Dict[str, Any] = None):
        """Log error with additional context information."""
        logger = self.get_logger(module_name)
        context = context or {}
        
        error_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'error_context': context,
            'exception_info': True
        }
        
        logger.error(f"Error occurred: {error}", extra=error_data, exc_info=True)

    def create_audit_logger(self, module_name: str = "audit") -> logging.Logger:
        """Create a special audit logger with specific formatting."""
        audit_config = {
            'level': 'INFO',
            'format': 'JSON'
        }
        
        # Add audit-specific configuration
        self.config['modules'][module_name] = audit_config
        return self.get_logger(module_name)

    def flush_all_loggers(self):
        """Flush all logger handlers."""
        for logger in self.loggers.values():
            for handler in logger.handlers:
                handler.flush()

    def close_all_loggers(self):
        """Close all logger handlers."""
        for logger in self.loggers.values():
            for handler in logger.handlers:
                handler.close()
            logger.handlers.clear()
        self.loggers.clear()

    def get_log_stats(self) -> Dict[str, Any]:
        """Get statistics about current logging setup."""
        stats = {
            'project_name': self.project_name,
            'active_loggers': len(self.loggers),
            'logger_names': list(self.loggers.keys()),
            'base_log_dir': self.base_log_dir,
            'config_summary': {
                'console_output': self.config.get('console_output'),
                'file_output': self.config.get('file_output'),
                'rotation_enabled': self.config.get('rotation', {}).get('enabled')
            }
        }
        return stats

    def update_config(self, new_config: Dict[str, Any]):
        """Update logger configuration and reinitialize loggers."""
        self.config.update(new_config)
        # Close existing loggers
        self.close_all_loggers()
        # Re-setup base directory
        self._setup_base_directory()


# Singleton pattern for global access
_global_logger_instance = None

def get_global_logger(project_name: str = None, config_path: str = None) -> UniversalLogger:
    """Get or create global logger instance."""
    global _global_logger_instance
    if _global_logger_instance is None:
        _global_logger_instance = UniversalLogger(config_path, project_name)
    return _global_logger_instance


# Context manager for temporary logging configuration
class LoggingContext:
    """Context manager for temporary logging configuration changes."""
    
    def __init__(self, logger_instance: UniversalLogger, temp_config: Dict[str, Any]):
        self.logger_instance = logger_instance
        self.temp_config = temp_config
        self.original_config = None
        
    def __enter__(self):
        self.original_config = self.logger_instance.config.copy()
        self.logger_instance.update_config(self.temp_config)
        return self.logger_instance
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger_instance.update_config(self.original_config)


# Performance monitoring decorator
def log_performance(logger_instance: UniversalLogger, module_name: str):
    """Decorator to log function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger_instance.log_performance(
                    module_name, 
                    func.__name__, 
                    duration,
                    function_args=len(args),
                    function_kwargs=len(kwargs)
                )
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger_instance.log_error_with_context(
                    module_name, 
                    e, 
                    {
                        'function': func.__name__,
                        'duration': duration,
                        'args_count': len(args),
                        'kwargs_count': len(kwargs)
                    }
                )
                raise
        return wrapper
    return decorator


if __name__ == "__main__":
    # Example usage
    logger_tracker = UniversalLogger(project_name="test_project")
    
    # Basic logging
    logger = logger_tracker.get_logger("main")
    logger.info("Application started")
    
    # Performance logging
    logger_tracker.log_performance("main", "initialization", 0.1234)
    
    # Error logging with context
    try:
        raise ValueError("Test error")
    except ValueError as e:
        logger_tracker.log_error_with_context("main", e, {"context": "testing"})
    
    # Audit logging
    audit_logger = logger_tracker.create_audit_logger()
    audit_logger.info("User login", extra={"user_id": "12345", "ip": "192.168.1.1"})
    
    # Get stats
    stats = logger_tracker.get_log_stats()
    print(f"Logger stats: {stats}")
    
    # Clean up
    logger_tracker.close_all_loggers()