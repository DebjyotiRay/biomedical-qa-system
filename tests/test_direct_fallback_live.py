#!/usr/bin/env python3
"""
Live test script for the direct fallback mechanism
with actual API calls and citation fetching
"""

import os
import sys
import json
import logging
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from biomed_qa_system_final.qa_handler import BiomedicalQAHandler
from biomed_qa_system_final.utils import load_config, load_environment_variables

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs/biomedical_qa.log"), mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Test the direct fallback functionality with live API calls"""
    # Load environment variables and config
    load_environment_variables()
    
    try:
        config = load_config("config.json")
        
        # Ensure use_llm_parent is set to False to test direct fallback
        if config.get("inference", {}).get("use_llm_parent", True) != False:
            logger.warning("use_llm_parent is not set to False in config. Setting it now for testing.")
            if "inference" not in config:
                config["inference"] = {}
            config["inference"]["use_llm_parent"] = False
        
        # Initialize QA handler with a custom setup method that bypasses model loading
        qa_handler = BiomedicalQAHandler(config)
        
        # Monkey-patch the setup method to avoid model loading
        # Since we're using direct fallback, we don't need the main model
        
        # Override the setup method to always return True and not load the model
        def bypass_setup(self):
            logger.info("Bypassing model loading for direct fallback test")
            return True
            
        # Replace the original setup method with our bypass
        qa_handler.setup = bypass_setup.__get__(qa_handler, BiomedicalQAHandler)
        
        # Set up the fallback clients
        qa_handler._setup_fallback_clients()
        
        # Call our bypassed setup method
        if not qa_handler.setup():
            logger.error("Failed to set up QA system")
            return 1
        
        # Test with the same question used in the standalone citations.py example
        test_question = "How many days will it take to aggregate parkinsons disease pathology in primary neurons?"
        print(f"\nTest Question: {test_question}")
        print("Processing...")
        
        start_time = time.time()
        result = qa_handler.answer_question(test_question, include_citations=True)
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*80)
        print(result["formatted_answer"])
        print("="*80)
        print(f"(Source: {result['answer_source']} | Processing Time: {elapsed_time:.2f}s)")
        
        # Check if it's using direct fallback
        if "direct_fallback" in result["answer_source"]:
            print("\n✅ Test PASSED: System is using direct fallback as expected")
        else:
            print(f"\n❌ Test FAILED: System is not using direct fallback. Source: {result['answer_source']}")
        
        # Check if citations were fetched
        if result["citations"] and len(result["citations"]) > 0:
            print(f"\n✅ Citation service working: Found {len(result['citations'])} citations")
        else:
            print("\n⚠️ No citations found, but this might be expected for some queries")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in test execution: {e}", exc_info=True)
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
