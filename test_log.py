#!/usr/bin/env python3
"""
Simple test to verify logging functionality
"""

import os
import sys
import logging
import time
import datetime

# Ensure log directory exists
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
log_file = os.path.join(log_dir, "biomedical_qa.log")

print(f"Current working directory: {os.getcwd()}")
print(f"Log directory: {log_dir}")
print(f"Log file path: {log_file}")
print(f"Log file exists: {os.path.exists(log_file)}")
print(f"Log directory exists: {os.path.exists(log_dir)}")

os.makedirs(log_dir, exist_ok=True)

# Configure logging with a direct file handler
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_logger")

# Remove existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Add a FileHandler directly to the logger
file_handler = logging.FileHandler(log_file, mode='a')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - TEST - %(message)s'))
logger.addHandler(file_handler)

# Also add a stream handler to see output
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - TEST - %(message)s'))
logger.addHandler(stream_handler)

# Test writing to the log
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
test_message = f"Test log entry at {current_time}"

logger.info(test_message)
logger.info("Writing to log file directly - should appear in logs/biomedical_qa.log")

# Force flush the handler
file_handler.flush()

print(f"Wrote test message to log: {test_message}")
print(f"Check {log_file} for the new log entries")

# Also try direct file writing
try:
    with open(log_file, "a") as f:
        f.write(f"\n{current_time} - DIRECT FILE WRITE - Test entry\n")
    print(f"Also wrote to log file directly with open()")
except Exception as e:
    print(f"Error directly writing to log file: {e}")
