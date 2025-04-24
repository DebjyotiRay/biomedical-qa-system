"""
Centralized logging configuration for the biomedical QA system
"""

import os
import logging
import sys

# Ensure log directory exists
logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Log file path
log_file = os.path.join(logs_dir, "biomedical_qa.log")

# Root logger configuration
def configure_logger(name=None):
    """Configure and return a logger with the proper handlers"""
    logger = logging.getLogger(name)
    
    # Only add handlers if they don't exist
    if not logger.handlers:
        # Set logging level
        logger.setLevel(logging.INFO)
        
        # Create handlers
        file_handler = logging.FileHandler(log_file, mode='a')
        console_handler = logging.StreamHandler()
        
        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Force handlers to flush after each log message
        file_handler.setLevel(logging.INFO)
        console_handler.setLevel(logging.INFO)
        
    return logger

# Configure the root logger
root_logger = configure_logger()

# Main function to get a module-specific logger
def get_logger(name):
    """Get a configured logger for the specified name"""
    return configure_logger(name)
