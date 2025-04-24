#!/usr/bin/env python3
"""
Test script specifically for citation integration in qa_handler
"""

import os
import sys
import logging
import time

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.qa_handler import BiomedicalQAHandler
from src.utils import load_config, load_environment_variables
from src.citations import fetch_pubmed_citations

# Configure logging
# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/biomedical_qa.log", mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_direct_citation():
    """Test citation fetching directly from citations.py"""
    print("\n===== TESTING DIRECT CITATION FETCHING =====")
    
    test_question = "How many days will it take to aggregate parkinsons disease pathology in primary neurons?"
    print(f"Query: {test_question}")
    
    start_time = time.time()
    try:
        # Use default parameters from config
        citations = fetch_pubmed_citations(
            test_question,
            max_citations=7,
            prioritize_reviews=True,
            max_age_years=5
        )
        
        print(f"\nFound {len(citations)} citations directly:")
        for i, citation in enumerate(citations, 1):
            print(f"{i}. {citation.title}")
            if citation.authors: print(f"   Authors: {citation.authors}")
            if citation.journal: print(f"   Journal: {citation.journal} ({citation.year})")
            if citation.pmid: print(f"   PMID: {citation.pmid}")
            print()
            
    except Exception as e:
        logger.error(f"Error in direct citation test: {e}", exc_info=True)
        print(f"Error fetching citations directly: {e}")
    
    elapsed_time = time.time() - start_time
    print(f"Direct citation fetching completed in {elapsed_time:.2f} seconds")

def test_qa_handler_citation():
    """Test citation fetching through qa_handler"""
    print("\n===== TESTING CITATION FETCHING VIA QA HANDLER =====")
    
    # Load environment variables and config
    load_environment_variables()
    
    try:
        config = load_config("src/config.json")
        
        # Ensure use_llm_parent is set to False to test direct fallback
        config["inference"]["use_llm_parent"] = False
        
        print("Initializing QA handler...")
        qa_handler = BiomedicalQAHandler(config)
        
        # Monkey-patch the setup method to bypass model loading
        def bypass_setup(self):
            logger.info("Bypassing model loading for citation test")
            return True
        qa_handler.setup = bypass_setup.__get__(qa_handler, BiomedicalQAHandler)
        
        # Set up the fallback clients
        qa_handler._setup_fallback_clients()
        qa_handler.setup()
        
        # Use the same test question
        test_question = "How many days will it take to aggregate parkinsons disease pathology in primary neurons?"
        print(f"\nTest Question: {test_question}")
        
        # Get just the citations part of the qa_handler process
        print("Fetching citations through qa_handler...")
        
        # A more direct way to test just the citation functionality
        try:
            pubmed_config = config.get("pubmed", {})
            start_time = time.time()
            
            # Extract this part from qa_handler.answer_question to test directly
            citations = fetch_pubmed_citations(
                test_question,
                max_citations=pubmed_config.get("max_citations", 3),
                prioritize_reviews=pubmed_config.get("prioritize_reviews", True),
                max_age_years=pubmed_config.get("max_age_years", 5)
            )
            
            elapsed_time = time.time() - start_time
            
            print(f"\nFound {len(citations)} citations through qa_handler path:")
            for i, citation in enumerate(citations, 1):
                print(f"{i}. {citation.title}")
                if citation.authors: print(f"   Authors: {citation.authors}")
                if citation.journal: print(f"   Journal: {citation.journal} ({citation.year})")
                if citation.pmid: print(f"   PMID: {citation.pmid}")
                print()
                
            print(f"Citation fetching via qa_handler completed in {elapsed_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error fetching citations via qa_handler path: {e}", exc_info=True)
            print(f"Error: {e}")
        
        # Now test the full qa_handler path
        print("\n===== TESTING FULL QA HANDLER WITH CITATIONS =====")
        try:
            start_time = time.time()
            result = qa_handler.answer_question(test_question, include_citations=True)
            elapsed_time = time.time() - start_time
            
            print(f"\nAnswer from fallback: {result['answer_source']}")
            
            if result["citations"] and len(result["citations"]) > 0:
                print(f"\nFound {len(result['citations'])} citations in full process:")
                for i, citation in enumerate(result["citations"], 1):
                    print(f"{i}. {citation['title']}")
                    print(f"   Authors: {citation.get('authors', 'N/A')}")
                    print(f"   Journal: {citation.get('journal', 'N/A')} ({citation.get('year', 'N/A')})")
                    print()
            else:
                print("\nNo citations found in full qa_handler process!")
                
            print(f"Full process completed in {elapsed_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error in full process: {e}", exc_info=True)
            print(f"Error: {e}")
        
    except Exception as e:
        logger.error(f"Error in test execution: {e}", exc_info=True)
        print(f"Error: {e}")

if __name__ == "__main__":
    # First test direct citation fetching
    test_direct_citation()
    
    # Then test through qa_handler
    test_qa_handler_citation()
