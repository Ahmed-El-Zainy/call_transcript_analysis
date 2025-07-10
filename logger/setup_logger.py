#!/usr/bin/env python3
"""
Project Setup Script for Universal Logger
This script helps you set up the universal logger in any project.
"""

import os
import sys
import argparse
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List

class LoggerProjectSetup:
    """Setup class for integrating universal logger into projects."""
    
    def __init__(self):
        self.project_types = {
            'web': {
                'description': 'Web applications (FastAPI, Django, Flask)',
                'modules': ['main', 'fastapi', 'django', 'flask', 'auth', 'database', 'api']
            },
            'ml': {
                'description': 'Machine Learning projects',
                'modules': ['main', 'data_processing', 'tensorflow', 'pytorch', 'sklearn', 'model_training']
            },
            'microservices': {
                'description': 'Microservices architecture',
                'modules': ['main', 'service_a', 'service_b', 'gateway', 'discovery', 'auth']
            },
            'etl': {
                'description': 'ETL (Extract, Transform, Load) pipelines',
                'modules': ['main', 'extract', 'transform', 'load', 'validation', 'scheduler']
            },
            'iot': {
                'description': 'IoT and real-time systems',
                'modules': ['main', 'sensor_data', 'actuator_control', 'network', 'device_manager']
            },
            'ecommerce': {
                'description': 'E-commerce platforms',
                'modules': ['main', 'user_management', 'product_catalog', 'order_processing', 'payment_gateway', 'inventory']
            },
            'devops': {
                'description': 'DevOps and infrastructure',
                'modules': ['main', 'server_monitoring', 'deployment', 'backup_system', 'metrics']
            },
            'batch': {
                'description': 'Batch processing systems',
                'modules': ['main', 'job_scheduler', 'worker', 'queue_manager', 'notification']
            },
            'custom': {
                'description': 'Custom project (you specify modules)',
                'modules': []
            }
        }
    
    def create_project_config(self, project_type: str, project_name: str, modules: List[str] = None, 
                            output_dir: str = ".", log_level: str = "INFO") -> Dict[str, Any]:
        """Create a project-specific configuration."""
        
        if modules is None:
            modules = self.project_types.get(project_type, {}).get('modules', [])
        
        config = {
            'base_log_dir': 'logs',
            'default_level': log_level,
            'console_output': True,
            'file_output': True,
            'hierarchical_structure': True,
            'format': 'STANDARD',
            'rotation': {
                'enabled': True,
                'type': 'size',
                'max_bytes': 10485760,
                'backup_count': 5,
                'compress': True
            },
            'context': {
                'add_hostname': True,
                'add_process_info': True,
                'add_thread_info': True,
                'custom_fields': {
                    'project_name': project_name,
                    'project_type': project_type,
                    'environment': 'development'
                }
            },
            'modules': {}
        }
        
        # Add module-specific configurations
        for module in modules:
            if module == 'audit':
                config['modules'][module] = {
                    'level': 'INFO',
                    'format': 'JSON'
                }
            elif module in ['database', 'auth', 'security']:
                config['modules'][module] = {
                    'level': 'INFO',
                    'format': 'DETAILED'
                }
            elif module in ['debug', 'test']:
                config['modules'][module] = {
                    'level': 'DEBUG',
                    'format': 'DETAILED'
                }
            else:
                config['modules'][module] = {
                    'level': log_level,
                    'format': 'STANDARD'
                }
        
        return config
    
    def create_logger_wrapper(self, project_name: str, output_dir: str = "."):
        """Create a project-specific logger wrapper."""
        wrapper_code = f'''"""
Project Logger Wrapper for {project_name}
This module provides easy access to the universal logger for this project.
"""

from enhanced_logger import UniversalLogger, get_global_logger, log_performance
import os
from pathlib import Path

# Get the directory of this file
CURRENT_DIR = Path(__file__).parent
CONFIG_PATH = CURRENT_DIR / "logging_config.yaml"

# Global logger instance for this project
_project_logger = None

def get_project_logger():
    """Get the global logger instance for this project."""
    global _project_logger
    if _project_logger is None:
        _project_logger = UniversalLogger(
            project_name="{project_name}",
            config_path=str(CONFIG_PATH) if CONFIG_PATH.exists() else None
        )
    return _project_logger

def get_logger(module_name: str):
    """Get a logger for a specific module in this project."""
    return get_project_logger().get_logger(module_name)

def log_performance_decorator(module_name: str):
    """Decorator for logging function performance."""
    return log_performance(get_project_logger(), module_name)

def log_error_with_context(module_name: str, error: Exception, context: dict = None):
    """Log error with context information."""
    return get_project_logger().log_error_with_context(module_name, error, context)

def create_audit_logger():
    """Create an audit logger for this project."""
    return get_project_logger().create_audit_logger()

def get_logger_stats():
    """Get statistics about the current logging setup."""
    return get_project_logger().get_log_stats()

def close_all_loggers():
    """Close all loggers (call this on application shutdown)."""
    global _project_logger
    if _project_logger:
        _project_logger.close_all_loggers()
        _project_logger = None

# Common logger instances for quick access
main_logger = get_logger("main")
'''
        
        wrapper_path = Path(output_dir) / f"{project_name}_logger.py"
        with open(wrapper_path, 'w') as f:
            f.write(wrapper_code)
        
        return wrapper_path
    
    def create_example_usage(self, project_name: str, project_type: str, 
                           modules: List[str], output_dir: str = "."):
        """Create example usage file for the project."""
        
        example_code = f'''"""
Example usage of the logger for {project_name}
This file shows how to use the logger in your {project_type} project.
"""

from {project_name}_logger import (
    get_logger, 
    log_performance_decorator, 
    log_error_with_context,
    create_audit_logger,
    main_logger
)
import time

def main():
    """Main example function."""
    # Basic logging
    main_logger.info("Application started")
    
    # Module-specific loggers
'''
        
        for module in modules[:3]:  # Show examples for first 3 modules
            example_code += f'''    {module}_logger = get_logger("{module}")
    {module}_logger.info("Module {module} initialized")
    
'''
        
        example_code += '''    # Performance logging with decorator
    @log_performance_decorator("main")
    def some_operation():
        time.sleep(0.1)  # Simulate work
        return "completed"
    
    result = some_operation()
    main_logger.info(f"Operation result: {result}")
    
    # Error logging with context
    try:
        raise ValueError("Example error")
    except ValueError as e:
        log_error_with_context("main", e, {"context": "example_usage"})
    
    # Audit logging
    audit_logger = create_audit_logger()
    audit_logger.info("User action", extra={
        "user_id": "user_123",
        "action": "login",
        "ip_address": "192.168.1.100"
    })

if __name__ == "__main__":
    main()
'''
        
        example_path = Path(output_dir) / f"{project_name}_example.py"
        with open(example_path, 'w') as f:
            f.write(example_code)
        
        return example_path
    
    def create_requirements_file(self, output_dir: str = "."):
        """Create requirements.txt file with necessary dependencies."""
        requirements = [
            "PyYAML>=6.0",
            "python-json-logger>=2.0.0",  # For JSON logging
        ]
        
        requirements_path = Path(output_dir) / "logger_requirements.txt"
        with open(requirements_path, 'w') as f:
            f.write('\n'.join(requirements))
        
        return requirements_path
    
    def setup_project(self, project_name: str, project_type: str, 
                     modules: List[str] = None, output_dir: str = ".", 
                     log_level: str = "INFO"):
        """Set up the complete logger configuration for a project."""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"Setting up logger for project: {project_name}")
        print(f"Project type: {project_type}")
        print(f"Output directory: {output_path.absolute()}")
        
        # Create configuration
        config = self.create_project_config(project_type, project_name, modules, output_dir, log_level)
        
        # Save configuration
        config_path = output_path / "logging_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print(f"✓ Created configuration file: {config_path}")
        
        # Create logger wrapper
        wrapper_path = self.create_logger_wrapper(project_name, output_dir)
        print(f"✓ Created logger wrapper: {wrapper_path}")
        
        # Create example usage
        example_path = self.create_example_usage(project_name, project_type, 
                                               modules or [], output_dir)
        print(f"✓ Created example usage: {example_path}")
        
        # Create requirements file
        req_path = self.create_requirements_file(output_dir)
        print(f"✓ Created requirements file: {req_path}")
        
        print(f"\nSetup complete! Files created in: {output_path.absolute()}")
        print(f"\nNext steps:")
        print(f"1. Install dependencies: pip install -r {req_path.name}")
        print(f"2. Copy enhanced_logger.py to your project directory")
        print(f"3. Import and use: from {project_name}_logger import get_logger")
        print(f"4. Run example: python {example_path.name}")
        
        return {
            'config_path': config_path,
            'wrapper_path': wrapper_path,
            'example_path': example_path,
            'requirements_path': req_path
        }

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Set up Universal Logger for your project")
    parser.add_argument("project_name", help="Name of your project")
    parser.add_argument("--type", choices=list(LoggerProjectSetup().project_types.keys()), 
                       default="custom", help="Type of project")
    parser.add_argument("--modules", nargs='+', help="List of modules to include")
    parser.add_argument("--output", default=".", help="Output directory")
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Default log level")
    parser.add_argument("--list-types", action="store_true", help="List available project types")
    
    args = parser.parse_args()
    
    setup = LoggerProjectSetup()
    
    if args.list_types:
        print("Available project types:")
        for ptype, info in setup.project_types.items():
            print(f"  {ptype}: {info['description']}")
            if info['modules']:
                print(f"    Default modules: {', '.join(info['modules'])}")
            print()
        return
    
    # Get modules
    modules = args.modules
    if not modules and args.type != 'custom':
        modules = setup.project_types[args.type]['modules']
    elif not modules:
        modules = ['main']
    
    # Set up the project
    try:
        setup.setup_project(args.project_name, args.type, modules, args.output, args.log_level)
    except Exception as e:
        print(f"Error setting up project: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()