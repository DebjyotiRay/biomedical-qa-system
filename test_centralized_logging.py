#!/usr/bin/env python3
"""
Test script for centralized logging
"""

import os
import sys
import time
import datetime

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our centralized logger
from src.logger_config import get_logger

# Get a logger for this module
logger = get_logger("test_centralized")

def main():
    """Test the centralized logging system"""
    print("Testing centralized logging configuration")
    
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    test_message = f"Centralized logging test at {current_time}"
    
    # Log at different levels
    logger.debug("This is a debug message (should not appear)")
    logger.info(test_message)
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    print(f"Logged test messages to logs/biomedical_qa.log")
    print(f"Check the log file to verify entries")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
